"""Difficulty-axis validation pass for IFBench.

Measurement only -- no optimization, no GEPA loop, no schedule arms,
no new experimental wiring. Reuses `src/ifbench_substrate` +
`src/ifbench_program` (seed) + `src/ifbench_feedback.metric_fn`
unchanged.

Scores the base seed program on the carved test split bucketed by
`constraint_count` (the integer count on every IFBench example, set
by the Chunk-11 loader). Cap per bucket: 25; if a bucket has fewer
than 25 available, use what exists and note the n.

Note: the Chunk-11 loader filters to `constraint_count >= 3` at
carve time. The carved test split therefore has data ONLY in
buckets 3, 4, 5. Buckets 1 and 2 are reported with `n=0`; the
monotonicity check applies to the buckets that have data.

Concurrency: `dspy.Evaluate(num_threads=48)` (de-risk pilot at 16
threads observed 0 throttle / 429 / backoff; we push toward 48).

Outputs under `results/difficulty_validation/`:
  - `per_example.json` : per-id score + constraint_count
  - `per_bucket.json`  : per-bucket {n, mean_score, std_score}
  - `summary.json`     : monotonicity flag + breaks + wall + provenance
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import dspy
import yaml
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.ifbench_substrate import (
    ifbench_substrate,
    load_ifbench_splits,
    verify_ifbench_difficulty_table_hash,
)
from src.run_gepa import load_lm_configs_from_env

OUT_DIR = REPO / "results" / "difficulty_validation"
PER_BUCKET_TARGET = 25
NUM_THREADS = 48
BUCKET_SAMPLE_SEED = 42


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

    task_lm_config, _ = load_lm_configs_from_env(config=config)
    print(
        f"task LM: {task_lm_config.model} @ temp {task_lm_config.temperature}; "
        f"dspy.Evaluate num_threads={NUM_THREADS}; cache=False (to exercise real concurrency)"
    )
    # Build the LM with cache=False so the 48-thread concurrency test
    # measures real HTTP fan-out (the Chunk-14 pilot's test eval already
    # cached responses for every test-split example at the seed
    # candidate; using LMConfig.to_lm() default cache=True would have
    # returned the cached samples in 1.6s and made the throttle scan
    # meaningless). Otherwise identical to LMConfig.to_lm().
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
    seed_instr = program.get_module_instruction("answer")
    print(f"seed instruction len={len(seed_instr)} chars; first 80 = {seed_instr[:80]!r}…")

    print("Loading carved splits (test = constraint_count >= 3, optimization-untouched)…")
    splits = load_ifbench_splits(config=config)
    test_split = splits[3]
    test_floor = min(int(ex["constraint_count"]) for ex in test_split)
    print(f"test split: n={len(test_split)}, min(constraint_count)={test_floor}")

    test_by_cc: dict[int, list] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for ex in test_split:
        cc = int(ex["constraint_count"])
        if cc in test_by_cc:
            test_by_cc[cc].append(ex)
    for cc in sorted(test_by_cc):
        print(f"  cc={cc}: {len(test_by_cc[cc])} available")

    rng = random.Random(BUCKET_SAMPLE_SEED)

    def _take(pool: list, n: int) -> list:
        if len(pool) <= n:
            return list(pool)
        return rng.sample(pool, n)

    buckets: dict[int, list] = {
        cc: _take(test_by_cc[cc], PER_BUCKET_TARGET) for cc in (1, 2, 3, 4, 5)
    }
    print("Sampled per bucket:")
    for cc in sorted(buckets):
        print(f"  bucket {cc}: drew {len(buckets[cc])} (target {PER_BUCKET_TARGET})")

    all_examples: list = []
    for cc in (1, 2, 3, 4, 5):
        all_examples.extend(buckets[cc])
    print(f"total examples to score: {len(all_examples)}")

    if not all_examples:
        print("ERROR: no examples to score; aborting.")
        return 1

    t0 = time.time()
    evaluator = dspy.Evaluate(
        devset=all_examples,
        metric=substrate.metric_fn,
        num_threads=NUM_THREADS,
        failure_score=0.0,
        provide_traceback=True,
        max_errors=len(all_examples) * 100,
    )
    res = evaluator(program)
    wall = time.time() - t0
    print(f"scoring wall: {wall:.1f}s")

    by_bucket: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    per_example: list[dict] = []
    for ex, _pred, score in res.results:
        cc = int(ex["constraint_count"])
        s = float(score)
        per_example.append({
            "id": ex["id"],
            "source_key": ex["source_key"],
            "constraint_count": cc,
            "score": s,
        })
        by_bucket[cc].append(s)

    per_bucket: list[dict] = []
    for cc in (1, 2, 3, 4, 5):
        scores = by_bucket[cc]
        per_bucket.append({
            "constraint_count": cc,
            "n": len(scores),
            "mean_score": (sum(scores) / len(scores)) if scores else None,
            "std_score": _population_std(scores) if scores else None,
        })

    # Monotonicity over buckets with n>0.
    means_present = [
        (b["constraint_count"], b["mean_score"]) for b in per_bucket if b["n"] > 0
    ]
    breaks: list[dict] = []
    is_monotonic = True
    for (cc_a, mean_a), (cc_b, mean_b) in zip(means_present, means_present[1:]):
        if not (mean_b < mean_a):
            is_monotonic = False
            breaks.append({
                "from_cc": cc_a,
                "to_cc": cc_b,
                "mean_from": mean_a,
                "mean_to": mean_b,
            })

    summary = {
        "n_examples_total": len(all_examples),
        "per_bucket_target": PER_BUCKET_TARGET,
        "num_threads": NUM_THREADS,
        "bucket_sample_seed": BUCKET_SAMPLE_SEED,
        "wall_time_s": round(wall, 2),
        "monotonic_decreasing_across_buckets_with_data": is_monotonic,
        "monotonicity_breaks": breaks,
        "buckets": per_bucket,
        "carved_test_split_size": len(test_split),
        "carved_test_split_min_cc": test_floor,
        "frozen_table_sha256": h,
        "task_model": task_lm_config.model,
        "task_temperature": task_lm_config.temperature,
    }
    (OUT_DIR / "per_example.json").write_text(json.dumps(per_example, indent=2))
    (OUT_DIR / "per_bucket.json").write_text(json.dumps(per_bucket, indent=2))
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 56)
    print(json.dumps(summary, indent=2))
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
