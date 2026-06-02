"""Score the base (seed-instruction) program on the full 100-instance
D_feedback once, and freeze the result as the difficulty table the Chunk-6
band samplers will read.

Resumable: per-id scores are flushed to ``results/d_feedback_scores.json``
after every example. A crash, kill, or rerun continues from the last id.
The final DifficultyTable is written to ``results/difficulty_table.json``
only after all 100 ids have a score.

Cost sanity: after the first three examples, prints observed wall-clock
and a rough cost estimate. If per-example wall-clock is more than 60s
sustained, aborts so an operator can investigate before burning $.

Run: ``.venv/bin/python -m src.score_d_feedback``

Required env: ``TASK_MODEL`` and the matching API key in ``.env``. The
reflection model is NOT used by this script (no GEPA loop, no reflection
calls). Only the task model is invoked, three times per example.
"""

from __future__ import annotations

import json
import os
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
from src.run_gepa import load_lm_configs_from_env  # noqa: E402

SCORES_PATH = REPO / "results" / "d_feedback_scores.json"
RECORDS_PATH = REPO / "results" / "d_feedback_records.json"
TABLE_PATH = REPO / "results" / "difficulty_table.json"
COST_CHECK_AT_N = 3
ABORT_IF_PER_EXAMPLE_S_OVER = 60.0
APPROX_TOKENS_PER_ROLLOUT = 1500  # 3 LM calls x ~500 tokens (heuristic)
QWEN_TOGETHER_USD_PER_1M = 0.20   # blended serverless price


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> int:
    load_dotenv(REPO / ".env")
    if not os.environ.get("TASK_MODEL"):
        print("ERROR: TASK_MODEL is not set; cannot score.")
        return 1

    config = yaml.safe_load((REPO / "config" / "experiment.yaml").read_text())
    task_lm_config, _ = load_lm_configs_from_env(config=config)

    print("Loading splits...")
    splits = load_splits(config=config)
    d_feedback = list(splits[0])
    expected_n = int(config["splits"]["d_feedback"])
    if len(d_feedback) != expected_n:
        print(f"ERROR: expected {expected_n} d_feedback, got {len(d_feedback)}")
        return 1
    print(f"  d_feedback has {len(d_feedback)} examples")

    # Configure LM and build the program with seed instructions.
    dspy.settings.configure(lm=task_lm_config.to_lm())
    program = build_program()

    scores: dict[str, float] = _load_json(SCORES_PATH)
    records: dict[str, dict] = _load_json(RECORDS_PATH)
    already_done = sum(1 for ex in d_feedback if ex["id"] in records)
    if already_done:
        print(f"Resuming: {already_done}/{len(d_feedback)} ids already scored")

    t_total = time.time()
    per_example_times: list[float] = []
    cost_checked = False

    for i, ex in enumerate(d_feedback):
        hid = ex["id"]
        if hid in records:
            continue

        t0 = time.time()
        try:
            out = program(**dict(ex.inputs()))
            f1 = float(module_metric(ex, out))
        except Exception as e:
            print(f"\nERROR at i={i} id={hid!r}: {type(e).__name__}: {e}")
            print("Saving partial scores and aborting.")
            _save_json(SCORES_PATH, scores)
            _save_json(RECORDS_PATH, records)
            return 2
        dt = time.time() - t0
        per_example_times.append(dt)

        scores[hid] = f1
        records[hid] = {
            "id": hid,
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "predicted_answer": str(getattr(out, "answer", "") or ""),
            "f1": f1,
        }
        _save_json(SCORES_PATH, scores)
        _save_json(RECORDS_PATH, records)
        done = sum(1 for e in d_feedback if e["id"] in records)
        print(
            f"  [{done:3d}/{len(d_feedback)}] id={hid}  F1={f1:.2f}  "
            f"({dt:.1f}s)"
        )

        # Cost sanity check after the first N examples.
        if not cost_checked and len(per_example_times) >= COST_CHECK_AT_N:
            cost_checked = True
            avg_s = sum(per_example_times) / len(per_example_times)
            max_s = max(per_example_times)
            approx_cost_per_ex = (
                APPROX_TOKENS_PER_ROLLOUT * QWEN_TOGETHER_USD_PER_1M / 1_000_000
            )
            est_remaining_cost = approx_cost_per_ex * (len(d_feedback) - done)
            print()
            print("== Cost / latency sanity check (first 3 examples) ==")
            print(f"  avg per-example wall-clock: {avg_s:.1f}s")
            print(f"  max per-example wall-clock: {max_s:.1f}s")
            print(
                f"  rough cost per example: ${approx_cost_per_ex:.4f}"
                f" (assuming ~{APPROX_TOKENS_PER_ROLLOUT} tok x"
                f" ${QWEN_TOGETHER_USD_PER_1M:.2f}/1M blended)"
            )
            print(f"  estimated remaining cost for {len(d_feedback) - done} more examples: ${est_remaining_cost:.2f}")
            if max_s > ABORT_IF_PER_EXAMPLE_S_OVER:
                print(
                    f"\nABORT: max example wall-clock {max_s:.1f}s exceeds "
                    f"{ABORT_IF_PER_EXAMPLE_S_OVER:.0f}s threshold."
                )
                return 3
            print("  Looks sane; continuing.\n")

    total_dt = time.time() - t_total
    print()
    print(f"All {len(d_feedback)} examples scored in {total_dt:.1f}s total.")

    # Build and write the frozen difficulty table.
    aligned_scores = [scores[ex["id"]] for ex in d_feedback]
    table = build_difficulty_table(aligned_scores)
    table.save(TABLE_PATH)
    print(f"Difficulty table written to {TABLE_PATH}")
    print(
        f"  band sizes: easy={len(table.ids('easy'))}, "
        f"mid={len(table.ids('mid'))}, hard={len(table.ids('hard'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
