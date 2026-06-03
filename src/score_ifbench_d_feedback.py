"""Score the IFBench base (seed-instruction) program on D_feedback at
temperature 0.6 (run conditions), build the frozen difficulty table via
the Chunk-13 continuity gate, and save the gate diagnostics alongside.

Resumable: per-id scores and per-id records are flushed to
``results/ifbench/d_feedback_scores.json`` and
``results/ifbench/d_feedback_records.json`` after every example. A
crash, kill, or rerun continues from the last id.

Cost sanity: after the first three examples, prints observed wall-clock
and a rough cost estimate. If per-example wall-clock is more than 90s
sustained, aborts so an operator can investigate before burning $.

Run: ``.venv/bin/python -m src.score_ifbench_d_feedback``

Required env: ``TASK_MODEL`` (set to ``together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo``
per BUILD_PLAN §5) and the matching API key in ``.env``. The reflection
model is NOT used by this script (no GEPA loop, no reflection calls).
Only the task model is invoked, once per example.
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

from src.difficulty import (  # noqa: E402
    build_chosen_table,
    difficulty_table_sha256,
)
from src.ifbench_data import carve_ifbench_splits  # noqa: E402
from src.ifbench_feedback import score_and_feedback  # noqa: E402
from src.ifbench_program import build_program  # noqa: E402
from src.run_gepa import load_lm_configs_from_env  # noqa: E402

OUT_DIR = REPO / "results" / "ifbench"
SCORES_PATH = OUT_DIR / "d_feedback_scores.json"
RECORDS_PATH = OUT_DIR / "d_feedback_records.json"
TABLE_PATH = OUT_DIR / "difficulty_table.json"
GATE_PATH = OUT_DIR / "continuity_gate.json"
HASH_PATH = OUT_DIR / "difficulty_table.sha256"

COST_CHECK_AT_N = 3
ABORT_IF_PER_EXAMPLE_S_OVER = 90.0
# IFBench produces longer, format-heavy outputs than HotpotQA: budget
# roughly 3000 tokens per single LM call (prompt + response).
APPROX_TOKENS_PER_ROLLOUT = 3000
QWEN_TOGETHER_USD_PER_1M = 0.20


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

    seed_splits = int(config["splits"]["seed_splits"])
    print(f"Carving IFBench splits at seed_splits={seed_splits}...")
    carve = carve_ifbench_splits(seed=seed_splits)
    d_feedback = list(carve.splits[0])
    expected_n = carve.sizes["d_feedback"]
    if len(d_feedback) != expected_n:
        print(f"ERROR: expected {expected_n} d_feedback, got {len(d_feedback)}")
        return 1
    print(f"  d_feedback has {len(d_feedback)} examples")
    print(f"  task model: {task_lm_config.model} (temperature={task_lm_config.temperature})")

    dspy.settings.configure(lm=task_lm_config.to_lm())
    program = build_program()

    scores: dict[str, float] = _load_json(SCORES_PATH)
    records: dict[str, dict] = _load_json(RECORDS_PATH)
    already_done = sum(1 for ex in d_feedback if str(ex["id"]) in records)
    if already_done:
        print(f"Resuming: {already_done}/{len(d_feedback)} ids already scored")

    t_total = time.time()
    per_example_times: list[float] = []
    cost_checked = False

    for i, ex in enumerate(d_feedback):
        hid = str(ex["id"])
        if hid in records:
            continue

        t0 = time.time()
        try:
            out = program(**dict(ex.inputs()))
            result = score_and_feedback(ex, out)
            score = float(result["score"])
        except Exception as e:
            print(f"\nERROR at i={i} id={hid!r}: {type(e).__name__}: {e}")
            print("Saving partial scores and aborting.")
            _save_json(SCORES_PATH, scores)
            _save_json(RECORDS_PATH, records)
            return 2
        dt = time.time() - t0
        per_example_times.append(dt)

        scores[hid] = score
        records[hid] = {
            "id": ex["id"],
            "source_key": ex["source_key"],
            "prompt": ex["prompt"],
            "instruction_id_list": list(ex["instruction_id_list"]),
            "kwargs_list": list(ex["kwargs_list"]),
            "constraint_count": int(ex["constraint_count"]),
            "response": str(getattr(out, "response", "") or ""),
            "score": score,
            "n_satisfied": len(result["satisfied"]),
            "n_violations": len(result["violations"]),
        }
        _save_json(SCORES_PATH, scores)
        _save_json(RECORDS_PATH, records)
        done = sum(1 for e in d_feedback if str(e["id"]) in records)
        print(
            f"  [{done:3d}/{len(d_feedback)}] id={ex['id']:>4}  "
            f"score={score:.4f}  ({dt:.1f}s)"
        )

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

    # Build the difficulty table via the continuity gate.
    aligned_scores = [scores[str(ex["id"])] for ex in d_feedback]
    table, gate = build_chosen_table(aligned_scores)
    table.save(TABLE_PATH)
    table_hash = difficulty_table_sha256(TABLE_PATH)
    HASH_PATH.write_text(table_hash + "\n")
    _save_json(GATE_PATH, gate)

    print()
    print(f"Difficulty table written to {TABLE_PATH}")
    print(f"  SHA-256: {table_hash}")
    print(f"  chosen binning: {gate['chosen_binning']}")
    print(
        f"  band sizes: easy={len(table.ids('easy'))}, "
        f"mid={len(table.ids('mid'))}, hard={len(table.ids('hard'))}"
    )
    print(f"  verdict: {gate['verdict']}  (frontier={gate['frontier_size_under_chosen']}, "
          f"min for GO={gate['frontier_min_for_go']})")
    print(f"Gate diagnostics: {GATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
